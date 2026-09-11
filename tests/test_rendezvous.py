import unittest

from mission.rendezvous import RendezvousProtocol


class RendezvousProtocolTests(unittest.TestCase):
    def test_rover_waits_until_every_ack_reaches_it(self) -> None:
        protocol = RendezvousProtocol(3, (5, 5))
        protocol.propose((20, 10))

        protocol.drone_rover_contact(0)
        protocol.drone_drone_contact(0, 1)

        # A proposal is knowledge, not yet the place where a reporting drone
        # assumes the stationary rover can be found.
        self.assertEqual(protocol.drone_endpoint(1), (5, 5))
        self.assertEqual(
            protocol.snapshot().drone_announcements[1].position,
            (20, 10),
        )
        self.assertFalse(protocol.can_depart((20, 10)))

        # Drone 1 carries its own and drone 0's acks back to the rover, but
        # drone 2 still has not received the endpoint.
        protocol.drone_rover_contact(1)
        self.assertFalse(protocol.can_depart((20, 10)))

        protocol.drone_drone_contact(1, 2)
        self.assertEqual(protocol.drone_endpoint(2), (5, 5))
        self.assertFalse(protocol.can_depart((20, 10)))

        # The universal set exists on drone 1/2, but is not authoritative
        # until one of them physically communicates it to the rover.
        protocol.drone_rover_contact(2)
        self.assertTrue(protocol.can_depart((20, 10)))
        self.assertEqual(protocol.drone_endpoint(2), (20, 10))

    def test_empty_confirmed_endpoint_falls_forward_to_known_proposal(self) -> None:
        protocol = RendezvousProtocol(2, (5, 5))
        protocol.propose((20, 10))
        protocol.drone_rover_contact(0)
        protocol.drone_drone_contact(0, 1)

        self.assertEqual(protocol.drone_endpoint(1), (5, 5))
        self.assertEqual(
            protocol.drone_missed_endpoint(1, (5, 5)),
            (20, 10),
        )
        self.assertEqual(protocol.drone_endpoint(1), (20, 10))

    def test_arrival_commits_only_the_acknowledged_proposal(self) -> None:
        protocol = RendezvousProtocol(2, (1, 1))
        protocol.propose((9, 9))
        protocol.drone_rover_contact(0)

        self.assertFalse(protocol.rover_arrived((9, 9)))
        protocol.drone_rover_contact(1)
        self.assertTrue(protocol.rover_arrived((9, 9)))
        self.assertEqual(protocol.snapshot().current.position, (9, 9))

    def test_newer_proposal_invalidates_old_ack_set(self) -> None:
        protocol = RendezvousProtocol(2, (0, 0))
        first = protocol.propose((3, 3))
        protocol.drone_rover_contact(0)
        second = protocol.propose((4, 4))

        self.assertGreater(second.epoch, first.epoch)
        self.assertFalse(protocol.can_depart((3, 3)))
        self.assertFalse(protocol.can_depart((4, 4)))


if __name__ == "__main__":
    unittest.main()
